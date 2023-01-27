package com.ysz.dm.lib.lang.juc.future

import com.ysz.dm.lib.lang.juc.execute.Nap
import com.ysz.dm.lib.lang.juc.timer.Timer
import java.util.concurrent.CompletableFuture

/**
 * @author carl
 * @since 2023-01-20 1:46 AM
 **/
class Machina(
    private val id: Int,
    var state: State = State.START
) {
    override fun toString(): String {
        val str = if (state == State.END) "complete" else state.name
        return "Machina:$id:$str"
    }

    companion object {
        fun work(m: Machina): Machina {
            if (m.state != State.END) {
                Nap(0.1)
                m.state = m.state.step()
            }
            println(m)
            return m;
        }
    }


    enum class State {
        START, ONE, TWO, THREE, END;

        fun step(): State {
            return if (this == END) END else values()[this.ordinal + 1]
        }
    }
}

fun main() {
    val timer = Timer()
    /*1. complete future 本身没啥用, 但是 cf 包装还是有点用的*/
    val cf = CompletableFuture
        .completedFuture(Machina(0))
        .thenApplyAsync { Machina.work(it) }
        .thenApplyAsync { Machina.work(it) }
        .thenApplyAsync { Machina.work(it) }
        .thenApplyAsync { Machina.work(it) }

    println(timer.duration())
    cf.join()
    println(timer.duration())
}
