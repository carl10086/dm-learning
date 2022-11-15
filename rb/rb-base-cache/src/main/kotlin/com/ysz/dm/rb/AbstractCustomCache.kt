package com.ysz.dm.rb

/**
 * @author carl
 * @create 2022-11-15 3:51 PM
 **/
abstract class AbstractCustomCache<K, V>(
    /*common cfg for all implementation*/
    val cfg: CustomCacheCfg
) : CustomCache<K, V> {
}