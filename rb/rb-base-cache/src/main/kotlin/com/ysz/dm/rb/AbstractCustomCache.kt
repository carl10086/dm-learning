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
        k: K, sorLoader: Function<K, V?>, penetrationProtect: Boolean, fallback: Function<K, V>?
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
            fallback
                ?.apply(k)
                ?: throw CustomCacheException("load failed , key :${k}", e)
        }
    }

    override fun multiGet(
        keys: Collection<K>,
        loader: Function<Collection<K>, Map<K, V>>,
        fallback: Function<K, V>?
    ): Map<K, V> {
        return try {
            if (keys.isNullOrEmpty()) return emptyMap()

            /*1. get from cache*/
            val fromCache = doMultiGet(keys)

            /*2. miss keys = all - result*/
            val miss = keys subtract fromCache.keys

            /*3. no miss keys , direct return*/
            if (miss.isNullOrEmpty()) return fromCache

            /*4. fetch miss from loader*/
            val fromSor = loader.apply(miss)

            if (fromSor.isNotEmpty()) doMultiPut(fromSor)

            /*5. merge results*/
            fromSor + fromCache
        } catch (e: Exception) {
            log.error("error occurred when multi get key:{}", keys)

            fallback
                ?.let { return buildMap { keys.forEach { put(it, fallback.apply(it)) } } }
                ?: throw CustomCacheException("multiGet failed, keys:${keys}")

        }
    }

//    override fun rm(k: K): Boolean {
//        TODO("Not yet implemented")
//    }

    /**
     * <pre>
     *     batch remove action.
     *
     *     it is recommended to override this method with bulk operations
     * </pre>
     */
    override fun rmAll(keys: Collection<K>): Map<K, Boolean> = keys.associateWith { this.rm(it) }

    /**
     * get from cache
     */
    abstract fun doGet(k: K): V?;

    /**
     * put into cache
     */
    abstract fun doPut(k: K, v: V);

    /**
     * multi get from cache
     */
    abstract fun doMultiGet(keys: Collection<K>): Map<K, V>

    /**
     * multi put into cache.  it's recommended to override by bulk operations
     */
    open fun doMultiPut(map: Map<K, V>) = map.forEach { (k, v) -> doPut(k, v) }


    companion object {
        private val log = LoggerFactory.getLogger(AbstractCustomCache::class.java)
    }
}