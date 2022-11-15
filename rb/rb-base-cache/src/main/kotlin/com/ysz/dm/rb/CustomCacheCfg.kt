package com.ysz.dm.rb

import java.time.Duration

/**
 * @author carl
 * @create 2022-11-15 3:55 PM
 **/
data class CustomCacheCfg constructor(
    /*auto add a prefix for key like namespace , such storage like redis support key prefix search .*/
    val keyPrefix: String = ":",
    /*logic name , maybe used for monitor*/
    val name: String,
    /*every key need a ttl cache after write data*/
    val ttl: Duration = Duration.ofMinutes(15)
)

