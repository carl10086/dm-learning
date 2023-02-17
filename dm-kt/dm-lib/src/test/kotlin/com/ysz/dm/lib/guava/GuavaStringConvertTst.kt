package com.ysz.dm.lib.guava

import com.google.common.base.CaseFormat
import com.ysz.dm.lib.common.atomictest.eq
import org.junit.jupiter.api.Test

/**
 * @author carl
 * @since 2023-02-17 9:17 PM
 **/
internal class GuavaStringConvertTst {

    @Test
    fun `test convert`() {
        val src = "helloWorldMyFriendDO"
        val dest = CaseFormat.LOWER_CAMEL.to(
            CaseFormat.LOWER_UNDERSCORE, src
        )
        println(
            dest
        )

        src eq CaseFormat.LOWER_UNDERSCORE.to(
            CaseFormat.LOWER_CAMEL, dest
        )
    }
}