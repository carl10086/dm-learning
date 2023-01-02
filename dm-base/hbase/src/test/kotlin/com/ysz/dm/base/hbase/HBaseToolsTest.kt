package com.ysz.dm.base.hbase

import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @since 2023/1/3
 */
internal class HBaseToolsTest {

    @Test
    fun `test doubleToBytes`() {
        checkDoubleArray(doubleArrayOf(1.0, 2.0, 3.0))
    }


    private fun checkDoubleArray(doubleArray: DoubleArray) {
        val byteArray = HBaseTools.doubleArrayToBytes(doubleArray)
        val array = HBaseTools.toDoubleArray(byteArray)
        Assertions.assertArrayEquals(
            doubleArray,
            array
        )
    }
}