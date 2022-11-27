package com.ysz.dm.rb.base.core.tools.sample

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @createAt 2022/11/25
 */
internal class AtomicIntegerSampleTest {
    @Test
    internal fun testSample() {
        for (i in 1..10) {
            println(AtomicIntegerSample.sample(10))
        }
    }
}