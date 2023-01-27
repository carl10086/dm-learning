package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-14 2:28 AM
 **/
internal class OperationsOnCollection {

    @Test
    fun `test CreatingLists`() {
        List(5) { it } eq listOf(0, 1, 2, 3, 4)
    }
}