package com.ysz.dm.rb.apigw.infra.web.security

import com.ysz.dm.rb.apigw.infra.web.security.CustomJwtFilter.Companion.BEARER
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test

/**
 * <pre>
 * class desc here
</pre> *
 * @author carl.yu
 * @createAt 2022/11/14
 */
internal class CustomJwtFilterTest {

    @Test
    internal fun testNotNull() {
        Assertions.assertTrue(invalid(null))
        Assertions.assertTrue(invalid("abadf"))
        Assertions.assertFalse(invalid("${BEARER}"))
    }


    private fun invalid(src: String?) = src?.startsWith(BEARER) != true
}