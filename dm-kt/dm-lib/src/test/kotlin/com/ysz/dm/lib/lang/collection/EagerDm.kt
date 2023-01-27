package com.ysz.dm.lib.lang.collection

import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-16 12:18 AM
 **/
internal class EagerDm {

    @Test
    fun `test eager`() {
        println(listOf(1, 2, 3).asSequence().map { it + 1 }.toList())
    }
}