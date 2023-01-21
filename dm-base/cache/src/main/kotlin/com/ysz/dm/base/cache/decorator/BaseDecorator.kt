package com.ysz.dm.base.cache.decorator

import com.ysz.dm.base.cache.CustomCache

/**
 * @author carl
 * @since 2023-01-21 4:10 PM
 **/
open class BaseDecorator<K, V>(private val cache: CustomCache<K, V>) : CustomCache<K, V> by cache