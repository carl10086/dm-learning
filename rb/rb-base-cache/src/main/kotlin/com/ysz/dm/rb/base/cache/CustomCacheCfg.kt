package com.ysz.dm.rb.base.cache

import java.time.Duration
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-15 3:55 PM
 **/
data class CustomCacheCfg<K, V> constructor(
    /*auto add a prefix for key like namespace , such storage like redis support key prefix search .*/
//    val keyPrefix: String = ":",
    /*every key need a ttl cache after write data*/
    val ttl: Duration = Duration.ofMinutes(15),
    /*fallback*/
    var fallback: Function<K, V>? = null,
    /*loader*/
    var loader: Function<K, V>,
    /*multi loader*/
    val multiLoader: Function<Collection<K>, Map<K, V>> = Function { keys ->
        buildMap {
            for (key in keys) {
                loader.apply(key)?.let { v -> put(key, v) }
            }
        }
    }
)

