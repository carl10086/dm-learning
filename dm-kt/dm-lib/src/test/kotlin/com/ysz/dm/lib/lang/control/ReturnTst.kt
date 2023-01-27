package com.ysz.dm.lib.lang.control

import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-01-21 10:22 PM
 **/
internal class ReturnTst {
    private fun f1(i: Int): Int? {
        return try {
            val v: Int? = try {
                if (i < 0) throw RuntimeException("i<0")
                i * 2
            } catch (e: Exception) {
                /*这里的 return 会直接结束 lambda*/
                return 0
            }
            println("1->ReturnTst:")
            v
        } catch (e: Exception) {
            -1
        }
    }

    @Test
    fun `test f1`() {
        println(f1(-2))
        println("----------")
        println(f1(2))
    }
}