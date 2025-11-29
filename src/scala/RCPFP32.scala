import chisel3._
import circt.stage.ChiselStage
import chisel3.util._
import fudian.{FCMA_ADD_s1, FCMA_ADD_s2, FMUL_s1, FMUL_s2, FMUL_s3, FMULToFADD, RawFloat}
import fudian.utils.Multiplier

object RCPFP32Parameters {
  val C0 = "h3F800000".U(32.W)  // 1.0
  val C1 = "hBF800000".U(32.W)  // -1.0
  val C2 = "h3F800000".U(32.W)  // 1.0
  
  val ZERO = 0.U(32.W)
  val INF  = "h7F800000".U(32.W)
  val NAN  = "h7FC00000".U(32.W)
}

object RCPFP32Utils {
  implicit class DecoupledPipe[T <: Data](val decoupledBundle: DecoupledIO[T]) extends AnyVal {
    def handshakePipeIf(en: Boolean): DecoupledIO[T] = {
      if (en) {
        val out = Wire(Decoupled(chiselTypeOf(decoupledBundle.bits)))
        val rValid = RegInit(false.B)
        val rBits  = Reg(chiselTypeOf(decoupledBundle.bits))
        decoupledBundle.ready  := !rValid || out.ready
        out.valid              := rValid
        out.bits               := rBits
        when(decoupledBundle.fire) {
          rBits  := decoupledBundle.bits
          rValid := true.B
        } .elsewhen(out.fire) {
          rValid := false.B
        }
        out
      } else {
        decoupledBundle
      }
    }
  }
}

import RCPFP32Utils._

class MULFP32[T <: Bundle](ctrlSignals: T) extends Module {
  val expWidth  = 8
  val precision = 24
  
  class InBundle extends Bundle {
    val a    = UInt(32.W)
    val b    = UInt(32.W)
    val rm   = UInt(3.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  class OutBundle extends Bundle {
    val result = UInt(32.W)
    val toAdd  = new FMULToFADD(expWidth, precision)
    val ctrl   = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  val mul   = Module(new Multiplier(precision + 1, pipeAt = Seq()))
  val mulS1 = Module(new FMUL_s1(expWidth, precision))
  val mulS2 = Module(new FMUL_s2(expWidth, precision))
  val mulS3 = Module(new FMUL_s3(expWidth, precision))
  
  mulS1.io.a  := io.in.bits.a
  mulS1.io.b  := io.in.bits.b
  mulS1.io.rm := io.in.bits.rm
  
  val rawA = RawFloat.fromUInt(io.in.bits.a, expWidth, precision)
  val rawB = RawFloat.fromUInt(io.in.bits.b, expWidth, precision)
  mul.io.a := rawA.sig
  mul.io.b := rawB.sig
  mul.io.regEnables.foreach(_ := true.B)
  
  val s1 = Wire(Decoupled(new Bundle {
    val mulS1Out = mulS1.io.out.cloneType
    val prod     = mul.io.result.cloneType
    val ctrl     = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s1Pipe = s1.handshakePipeIf(true)
  s1.valid         := io.in.valid
  s1.bits.mulS1Out := mulS1.io.out
  s1.bits.prod     := mul.io.result
  s1.bits.ctrl     := io.in.bits.ctrl
  io.in.ready      := s1.ready
  
  mulS2.io.in   := s1Pipe.bits.mulS1Out
  mulS2.io.prod := s1Pipe.bits.prod
  
  val s2 = Wire(Decoupled(new Bundle {
    val mulS2Out = mulS2.io.out.cloneType
    val ctrl     = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s2Pipe = s2.handshakePipeIf(true)
  s2.valid         := s1Pipe.valid
  s2.bits.mulS2Out := mulS2.io.out
  s2.bits.ctrl     := s1Pipe.bits.ctrl
  s1Pipe.ready     := s2.ready
  
  mulS3.io.in := s2Pipe.bits.mulS2Out
  
  val s3     = Wire(Decoupled(new OutBundle))
  val s3Pipe = s3.handshakePipeIf(true)
  s3.valid          := s2Pipe.valid
  s3.bits.result    := mulS3.io.result
  s3.bits.toAdd     := mulS3.io.to_fadd
  s3.bits.ctrl      := s2Pipe.bits.ctrl
  s2Pipe.ready      := s3.ready
  
  io.out <> s3Pipe
}

class CMAFP32[T <: Bundle](ctrlSignals: T) extends Module {
  val expWidth  = 8
  val precision = 24
  
  class InBundle extends Bundle {
    val a    = UInt(32.W)
    val b    = UInt(32.W)
    val c    = UInt(32.W)
    val rm   = UInt(3.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  class OutBundle extends Bundle {
    val result = UInt(32.W)
    val ctrl   = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  class MULToADD extends Bundle {
    val c       = UInt(32.W)
    val topCtrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val mul   = Module(new MULFP32[MULToADD](new MULToADD))
  val addS1 = Module(new FCMA_ADD_s1(expWidth, precision * 2, precision))
  val addS2 = Module(new FCMA_ADD_s2(expWidth, precision * 2, precision))
  
  mul.io.in.valid             := io.in.valid
  mul.io.in.bits.a            := io.in.bits.a
  mul.io.in.bits.b            := io.in.bits.b
  mul.io.in.bits.rm           := io.in.bits.rm
  mul.io.in.bits.ctrl.c       := io.in.bits.c
  mul.io.in.bits.ctrl.topCtrl := io.in.bits.ctrl
  io.in.ready                 := mul.io.in.ready
  
  addS1.io.a             := Cat(mul.io.out.bits.ctrl.c, 0.U(precision.W))
  addS1.io.b             := mul.io.out.bits.toAdd.fp_prod.asUInt
  addS1.io.b_inter_valid := true.B
  addS1.io.b_inter_flags := mul.io.out.bits.toAdd.inter_flags
  addS1.io.rm            := mul.io.out.bits.toAdd.rm
  
  val s4 = Wire(Decoupled(new Bundle {
    val out  = addS1.io.out.cloneType
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }))
  val s4Pipe = s4.handshakePipeIf(true)
  s4.valid         := mul.io.out.valid
  s4.bits.out      := addS1.io.out
  s4.bits.ctrl     := mul.io.out.bits.ctrl.topCtrl
  mul.io.out.ready := s4.ready
  
  addS2.io.in := s4Pipe.bits.out
  
  val s5     = Wire(Decoupled(new OutBundle))
  val s5Pipe = s5.handshakePipeIf(true)
  s5.valid       := s4Pipe.valid
  s5.bits.result := addS2.io.result
  s5.bits.ctrl   := s4Pipe.bits.ctrl
  s4Pipe.ready   := s5.ready
  
  io.out <> s5Pipe
}

class FilterFP32[T <: Bundle](ctrlSignals: Bundle) extends Module {
  class InBundle extends Bundle {
    val in   = UInt(32.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  class OutBundle extends Bundle {
    val out       = UInt(32.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val ctrl      = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  val s = io.in.bits.in(31)
  val e = io.in.bits.in(30, 23)
  val f = io.in.bits.in(22, 0)
  
  val isZero   = (e === 0.U) && (f === 0.U)
  val isInf    = (e === "hFF".U) && (f === 0.U)
  val isNaN    = (e === "hFF".U) && (f =/= 0.U)
  
  val bypass = isZero || isInf || isNaN
  
  val bypassVal = Wire(UInt(32.W))
  when (isNaN) {
    bypassVal := RCPFP32Parameters.NAN
  }.elsewhen (isZero) {
    bypassVal := Cat(s, RCPFP32Parameters.INF(30, 0))
  }.elsewhen (isInf) {
    bypassVal := Cat(s, RCPFP32Parameters.ZERO(30, 0))
  }.otherwise {
    bypassVal := RCPFP32Parameters.ZERO
  }
  
  val s1 = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)
  
  s1.valid          := io.in.valid
  s1.bits.out       := io.in.bits.in
  s1.bits.bypass    := bypass
  s1.bits.bypassVal := bypassVal
  s1.bits.ctrl      := io.in.bits.ctrl
  io.in.ready       := s1.ready
  
  io.out <> s1Pipe
}

// ===== Stage 1: 分解 mantissa 为 mHigh 和 mLow =====
class DecomposeRCP[T <: Bundle](ctrlSignals: T) extends Module {
  class InBundle extends Bundle {
    val x    = UInt(32.W)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  class OutBundle extends Bundle {
    val mHigh = UInt(7.W)   // mantissa 高 7 位
    val mLow  = UInt(32.W)  // mantissa 低 16 位转 FP32
    val exp   = UInt(8.W)   // 原始指数
    val sign  = Bool()      // 符号位
    val ctrl  = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  val expWidth  = 8
  val precision = 24
  val raw = RawFloat.fromUInt(io.in.bits.x, expWidth, precision)
  
  val sign = raw.sign
  val exp  = raw.exp
  val sig  = raw.sig
  
  // 提取 mHigh（sig 的高 7 位）
  val mHigh = sig(22, 16)
  
  // 提取 mLow（sig 的低 16 位）并转换为 FP32
  val mLowBits = Cat(sig(15, 0), 0.U(7.W))
  
  val mLowIsZero = mLowBits === 0.U
  val mLowLzd    = PriorityEncoder(Reverse(mLowBits))
  val mLowExp    = Mux(mLowIsZero, 0.U(8.W), (119.U(8.W) - mLowLzd)(7, 0))
  val mLowMant   = Mux(mLowIsZero, 0.U(23.W), (mLowBits << (mLowLzd + 1.U))(22, 0))
  val mLowFP32   = Cat(0.U(1.W), mLowExp, mLowMant)
  
  val s1 = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)
  
  s1.valid      := io.in.valid
  s1.bits.mHigh := mHigh
  s1.bits.mLow  := mLowFP32
  s1.bits.exp   := exp
  s1.bits.sign  := sign
  s1.bits.ctrl  := io.in.bits.ctrl
  io.in.ready   := s1.ready
  
  io.out <> s1Pipe
}

// ===== Stage 2: LUT 独立类 =====
class RCPLUT[T <: Bundle](ctrlSignals: T) extends Module {
  class InBundle extends Bundle {
    val mHigh = UInt(7.W)
    val ctrl  = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  class OutBundle extends Bundle {
    val T    = UInt(32.W)  // 1/(1 + mHigh/128)
    val ctrl = ctrlSignals.cloneType.asInstanceOf[T]
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  // LUT: 存储 1/(1 + i/128) for i = 0..127
  val table = VecInit((0 until 128).map { i =>
    val xi = 1.0 + i.toDouble / 128.0
    val yi = 1.0 / xi
    val bits = java.lang.Float.floatToIntBits(yi.toFloat)
    bits.U(32.W)
  })
  
  val T = table(io.in.bits.mHigh)
  
  val s1 = Wire(Decoupled(new OutBundle))
  val s1Pipe = s1.handshakePipeIf(true)
  
  s1.valid     := io.in.valid
  s1.bits.T    := T
  s1.bits.ctrl := io.in.bits.ctrl
  io.in.ready  := s1.ready
  
  io.out <> s1Pipe
}

class RCPFP32 extends Module {
  class InBundle extends Bundle {
    val in = UInt(32.W)
    val rm = UInt(3.W)
  }
  
  class OutBundle extends Bundle {
    val out = UInt(32.W)
  }
  
  val io = IO(new Bundle {
    val in  = Flipped(Decoupled(new InBundle))
    val out = Decoupled(new OutBundle)
  })
  
  // ===== Stage 0: Filter =====
  class FilterToDecompose extends Bundle {
    val rm = UInt(3.W)
  }
  
  val filter = Module(new FilterFP32[FilterToDecompose](new FilterToDecompose))
  io.in.ready               := filter.io.in.ready
  filter.io.in.valid        := io.in.valid
  filter.io.in.bits.in      := io.in.bits.in
  filter.io.in.bits.ctrl.rm := io.in.bits.rm
  
  // ===== Stage 1: Decompose =====
  class DecomposeToLUT extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
  }
  
  val decompose = Module(new DecomposeRCP[DecomposeToLUT](new DecomposeToLUT))
  filter.io.out.ready             := decompose.io.in.ready
  decompose.io.in.valid           := filter.io.out.valid
  decompose.io.in.bits.x          := filter.io.out.bits.out
  decompose.io.in.bits.ctrl.rm    := filter.io.out.bits.ctrl.rm
  decompose.io.in.bits.ctrl.bypass    := filter.io.out.bits.bypass
  decompose.io.in.bits.ctrl.bypassVal := filter.io.out.bits.bypassVal
  
  // ===== Stage 2: LUT =====
  class LUTToMUL extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val mLow      = UInt(32.W)
    val exp       = UInt(8.W)
    val sign      = Bool()
  }
  
  val lut = Module(new RCPLUT[LUTToMUL](new LUTToMUL))
  decompose.io.out.ready          := lut.io.in.ready
  lut.io.in.valid                 := decompose.io.out.valid
  lut.io.in.bits.mHigh            := decompose.io.out.bits.mHigh
  lut.io.in.bits.ctrl.rm          := decompose.io.out.bits.ctrl.rm
  lut.io.in.bits.ctrl.bypass      := decompose.io.out.bits.ctrl.bypass
  lut.io.in.bits.ctrl.bypassVal   := decompose.io.out.bits.ctrl.bypassVal
  lut.io.in.bits.ctrl.mLow        := decompose.io.out.bits.mLow
  lut.io.in.bits.ctrl.exp         := decompose.io.out.bits.exp
  lut.io.in.bits.ctrl.sign        := decompose.io.out.bits.sign
  
  // ===== Stage 3: MUL 计算 A = mLow * T =====
  class MULToCMA0 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val exp       = UInt(8.W)
    val sign      = Bool()
    val T         = UInt(32.W)
  }
  
  val mulA = Module(new MULFP32[MULToCMA0](new MULToCMA0))
  lut.io.out.ready               := mulA.io.in.ready
  mulA.io.in.valid               := lut.io.out.valid
  mulA.io.in.bits.a              := lut.io.out.bits.ctrl.mLow
  mulA.io.in.bits.b              := lut.io.out.bits.T
  mulA.io.in.bits.rm             := lut.io.out.bits.ctrl.rm
  mulA.io.in.bits.ctrl.rm        := lut.io.out.bits.ctrl.rm
  mulA.io.in.bits.ctrl.bypass    := lut.io.out.bits.ctrl.bypass
  mulA.io.in.bits.ctrl.bypassVal := lut.io.out.bits.ctrl.bypassVal
  mulA.io.in.bits.ctrl.exp       := lut.io.out.bits.ctrl.exp
  mulA.io.in.bits.ctrl.sign      := lut.io.out.bits.ctrl.sign
  mulA.io.in.bits.ctrl.T         := lut.io.out.bits.T
  
  // ===== Stage 4: CMA0 计算 tmp = A*1 + (-1) = A - 1 =====
  class CMA0ToCMA1 extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val exp       = UInt(8.W)
    val sign      = Bool()
    val T         = UInt(32.W)
    val A         = UInt(32.W)
  }
  
  val cma0 = Module(new CMAFP32[CMA0ToCMA1](new CMA0ToCMA1))
  mulA.io.out.ready              := cma0.io.in.ready
  cma0.io.in.valid               := mulA.io.out.valid
  cma0.io.in.bits.a              := mulA.io.out.bits.result  // A
  cma0.io.in.bits.b              := RCPFP32Parameters.C2     // 1.0
  cma0.io.in.bits.c              := RCPFP32Parameters.C1     // -1.0
  cma0.io.in.bits.rm             := mulA.io.out.bits.ctrl.rm
  cma0.io.in.bits.ctrl.rm        := mulA.io.out.bits.ctrl.rm
  cma0.io.in.bits.ctrl.bypass    := mulA.io.out.bits.ctrl.bypass
  cma0.io.in.bits.ctrl.bypassVal := mulA.io.out.bits.ctrl.bypassVal
  cma0.io.in.bits.ctrl.exp       := mulA.io.out.bits.ctrl.exp
  cma0.io.in.bits.ctrl.sign      := mulA.io.out.bits.ctrl.sign
  cma0.io.in.bits.ctrl.T         := mulA.io.out.bits.ctrl.T
  cma0.io.in.bits.ctrl.A         := mulA.io.out.bits.result
  
  // ===== Stage 5: CMA1 计算 r = A*tmp + 1 =====
  class CMA1ToMUL extends Bundle {
    val rm        = UInt(3.W)
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val exp       = UInt(8.W)
    val sign      = Bool()
    val T         = UInt(32.W)
  }
  
  val cma1 = Module(new CMAFP32[CMA1ToMUL](new CMA1ToMUL))
  cma0.io.out.ready              := cma1.io.in.ready
  cma1.io.in.valid               := cma0.io.out.valid
  cma1.io.in.bits.a              := cma0.io.out.bits.ctrl.A    // A
  cma1.io.in.bits.b              := cma0.io.out.bits.result    // A-1
  cma1.io.in.bits.c              := RCPFP32Parameters.C0       // 1.0
  cma1.io.in.bits.rm             := cma0.io.out.bits.ctrl.rm
  cma1.io.in.bits.ctrl.rm        := cma0.io.out.bits.ctrl.rm
  cma1.io.in.bits.ctrl.bypass    := cma0.io.out.bits.ctrl.bypass
  cma1.io.in.bits.ctrl.bypassVal := cma0.io.out.bits.ctrl.bypassVal
  cma1.io.in.bits.ctrl.exp       := cma0.io.out.bits.ctrl.exp
  cma1.io.in.bits.ctrl.sign      := cma0.io.out.bits.ctrl.sign
  cma1.io.in.bits.ctrl.T         := cma0.io.out.bits.ctrl.T
  
  // ===== Stage 6: MUL 计算 1/m = T * r =====
  class MULToAdjust extends Bundle {
    val bypass    = Bool()
    val bypassVal = UInt(32.W)
    val exp       = UInt(8.W)
    val sign      = Bool()
  }
  
  val mulFinal = Module(new MULFP32[MULToAdjust](new MULToAdjust))
  cma1.io.out.ready                  := mulFinal.io.in.ready
  mulFinal.io.in.valid               := cma1.io.out.valid
  mulFinal.io.in.bits.a              := cma1.io.out.bits.ctrl.T
  mulFinal.io.in.bits.b              := cma1.io.out.bits.result
  mulFinal.io.in.bits.rm             := cma1.io.out.bits.ctrl.rm
  mulFinal.io.in.bits.ctrl.bypass    := cma1.io.out.bits.ctrl.bypass
  mulFinal.io.in.bits.ctrl.bypassVal := cma1.io.out.bits.ctrl.bypassVal
  mulFinal.io.in.bits.ctrl.exp       := cma1.io.out.bits.ctrl.exp
  mulFinal.io.in.bits.ctrl.sign      := cma1.io.out.bits.ctrl.sign
  
  // ===== Stage 7: 调整指数并恢复符号 =====
  val rcpM     = mulFinal.io.out.bits.result
  val origExp  = mulFinal.io.out.bits.ctrl.exp
  val origSign = mulFinal.io.out.bits.ctrl.sign
  
  val rcpExp  = rcpM(30, 23)
  val rcpFrac = rcpM(22, 0)
  
  // 1/x = (1/m) * 2^(127-e)
  // finalExp = 254 - origExp
  val finalExp = (127.U - origExp + rcpExp)(7, 0)
  
  val adjustedResult = Cat(origSign, finalExp, rcpFrac)
  
  val result = Mux(mulFinal.io.out.bits.ctrl.bypass, 
                   mulFinal.io.out.bits.ctrl.bypassVal, 
                   adjustedResult)
  
  val sFinal = Wire(Decoupled(new OutBundle))
  val sFinalPipe = sFinal.handshakePipeIf(true)
  
  sFinal.valid     := mulFinal.io.out.valid
  sFinal.bits.out  := result
  mulFinal.io.out.ready := sFinal.ready
  
  io.out <> sFinalPipe
}

object RCPFP32Gen extends App {
  ChiselStage.emitSystemVerilogFile(
    new RCPFP32,
    Array("--target-dir", "rtl"),
    Array("-lowering-options=disallowLocalVariables")
  )
}
