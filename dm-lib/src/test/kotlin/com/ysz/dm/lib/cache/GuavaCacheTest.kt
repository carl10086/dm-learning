package com.ysz.dm.lib.cache

import com.google.common.cache.CacheBuilder
import com.google.common.cache.CacheLoader
import org.junit.jupiter.api.Test

/**
 * <pre>
 *      https://www.baeldung.com/guava-cache
 * </pre>
 *
 * @author carl
 * @create 2022-11-17 5:54 PM
 **/
internal class GuavaCacheTest {

    @Test
    internal fun `test_cacheLoader`() {
        val loader = object : CacheLoader<String, String>() {
            override fun load(key: String): String = key.uppercase()
        }


        val loadingCache = CacheBuilder.newBuilder().build(loader)

    }
}