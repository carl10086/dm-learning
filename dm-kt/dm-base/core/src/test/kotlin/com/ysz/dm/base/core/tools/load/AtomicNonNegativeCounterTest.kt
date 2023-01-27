package com.ysz.dm.base.core.tools.load

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2022/12/24
 */
internal class AtomicNonNegativeCounterTest {

    @Test
    fun `test incr`() {

        for (i in 1..1_000) {
            Assertions.assertTrue(AtomicNonNegativeCounter.incr() >= 0)
        }

    }
}