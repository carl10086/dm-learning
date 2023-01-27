package com.ysz.dm.lib.cache

import com.github.benmanes.caffeine.cache.Caffeine
import com.github.benmanes.caffeine.cache.LoadingCache
import org.junit.jupiter.api.Test
import java.time.Duration

/**
 * @author carl
 * @create 2022-11-17 6:06 PM
 **/
internal class CaffeineCacheTest {


    @Test
    internal fun test_loadCache() {
        val graphs: LoadingCache<String, String> = Caffeine.newBuilder()
            .maximumSize(10000)
            .expireAfterWrite(Duration.ofMinutes(5))
            .refreshAfterWrite(Duration.ofMinutes(1))
            .build { it ->
                it.lowercase()
            }
    }
}