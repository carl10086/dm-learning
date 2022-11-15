package com.ysz.dm.rb

import org.junit.jupiter.api.Test
import org.slf4j.LoggerFactory

/**
 * @author carl
 * @create 2022-11-15 4:51 PM
 */
internal class CustomCacheCfgTest {

    @Test
    internal fun `tst_Build`() {
        var user = CustomCacheCfg(name = "default")
//        user.username = "bb"
        log.info("user:{}", user)
    }

    companion object {
        private val log = LoggerFactory.getLogger(CustomCacheCfgTest::class.java)
    }
}