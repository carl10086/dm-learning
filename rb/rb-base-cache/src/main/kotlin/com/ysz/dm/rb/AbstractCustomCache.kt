package com.ysz.dm.rb

import org.slf4j.LoggerFactory
import java.util.function.Function

/**
 * @author carl
 * @create 2022-11-15 3:51 PM
 **/
abstract class AbstractCustomCache<K, V>(/*common cfg for all implementation*/
                                         val cfg: CustomCacheCfg
) : CustomCache<K, V> {

    override fun get(
        k: K, sorLoader: Function<K, V>, penetrationProtect: Boolean, fallback: Function<K, V>?
    ): V? {
        return try {/*1. success direct return*/
            doGet(k) ?: let {
                /*2. cache miss load from sor*/
                return sorLoader.apply(k)?.let {
                    doPut(k, it)
                    return it
                }
            }
        } catch (e: Exception) {
            log.error("error occurred when get key, k:{}", k, e)
            /*3. error happened, if fallback is null then wrap with custom exception*/
            fallback?.apply(k) ?: throw CustomCacheException("load failed , key :${k}", e)
        }
    }

    override fun multiGet(keys: Collection<K>, loader: Function<Collection<K>, Map<K, V>>, fallback: Function<K, V>?) {
        TODO("Not yet implemented")
    }

    override fun rm(k: K): Boolean {
        TODO("Not yet implemented")
    }

    override fun rmAll(keys: Collection<K>): Map<K, Boolean> {
        return super.rmAll(keys)
    }

    /**
     * get from cache
     */
    abstract fun doGet(k: K): V?;


    abstract fun doPut(k: K, v: V);


    companion object {
        private val log = LoggerFactory.getLogger(AbstractCustomCache::class.java)
    }
}