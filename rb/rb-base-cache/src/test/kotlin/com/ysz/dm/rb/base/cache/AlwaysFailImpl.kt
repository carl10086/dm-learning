package com.ysz.dm.rb.base.cache

/**
 * @author carl
 * @create 2022-11-17 9:00 PM
 **/
class AlwaysFailImpl : CustomCache<String, String> {
    override fun get(key: String): String? = throw CustomCacheException("unsupported")

    override fun multiGet(keys: Collection<String>): Map<String, String> = throw CustomCacheException("unsupported")

    override fun invalidate(k: String) = throw CustomCacheException("unsupported")

    override fun multiInvalidate(keys: Collection<String>) = throw CustomCacheException("unsupported")

    override fun put(k: String, v: String) = throw CustomCacheException("unsupported")
}