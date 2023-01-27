package com.ysz.dm.lib.lang.collection

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-13 2:35 AM
 **/
internal class LoopDm {

    @Test
    fun `test  labels`() {
        val strings = mutableListOf<String>()

        outer@ for (c in 'a'..'e') {
            for (i in 1..9) {
                when {
                    i == 5 -> continue@outer
                    "$c$i" == "c3" -> break@outer
                    else -> strings.add("$c$i")
                }
            }
        }

        strings eq listOf("a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4", "c1", "c2")
    }
}