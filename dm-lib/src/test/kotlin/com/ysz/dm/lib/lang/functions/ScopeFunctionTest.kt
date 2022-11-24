package com.ysz.dm.lib.lang.functions

import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory

/**
 * @author carl
 * @create 2022-11-21 5:42 PM
 **/
internal class ScopeFunctionTest {


    data class User(val name: String)



    @Test
    internal fun `test_runAndApply`() {

        val s1 = "S1"

        /*u can only change  object value, like return this */
        log.info("s1 to lowercase:{}", s1.apply { uppercase() })

        /**/
        log.info("s1 to lowercase:{}", s1.run(String::lowercase))


        /**/

        log.info("res:{}", mayBeNull(1) ?.let {  })
    }


    fun mayBeNull(a: Int): String? = if (a == 1) null else a.toString()


    companion object {
        private val log = LoggerFactory.getLogger(ScopeFunctionTest::class.java)
    }
}