package com.ysz.dm.rb.base.cache

/**
 * @author carl
 * @create 2022-11-17 9:54 PM
 **/
abstract class BaseCustomCache<K, V>(val cfg: CustomCacheCfg<K,V>) : CustomCache<K, V> {
}