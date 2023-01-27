package com.ysz.dm.lib.lang.functions

import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@since 2023/1/12
 **/
internal class ExtendFunctionsTest {

    /*扩展了 string 方法， 而且是 this*/
    fun String.singleQuote() = "'$this'"

    @Test
    fun `test singleQuote`() {
        "Hi".singleQuote() eq ""
    }
}