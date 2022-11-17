package com.ysz.dm.rb.base.cache.decorators

import com.ysz.dm.rb.base.cache.CustomCache

/**
 * @author carl
 * @create 2022-11-17 8:36 PM
 **/
abstract class BaseDecorator<K, V>(val cache: CustomCache<K, V>) : CustomCache<K, V> by cache