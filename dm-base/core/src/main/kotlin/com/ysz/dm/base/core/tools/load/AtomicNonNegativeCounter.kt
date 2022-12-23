package com.ysz.dm.base.core.tools.load

import java.util.concurrent.atomic.AtomicInteger

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2022/12/24
 **/
class AtomicNonNegativeCounter {

    companion object {
        private val counter = AtomicInteger(Int.MAX_VALUE - 5)

        fun incr(): Int = counter.updateAndGet { if (it < Int.MAX_VALUE) it + 1 else 0 }
    }
}