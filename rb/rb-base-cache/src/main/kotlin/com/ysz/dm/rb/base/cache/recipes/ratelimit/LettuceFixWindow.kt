package com.ysz.dm.rb.base.cache.recipes.ratelimit

import com.ysz.dm.rb.base.cache.impl.lettuce.CustomLettuceConnection

/**
 *
 * @author carl
 * @create 2022-11-21 12:03 PM
 **/
class LettuceFixWindow(
    val conn: CustomLettuceConnection<String, Int>
) {

}