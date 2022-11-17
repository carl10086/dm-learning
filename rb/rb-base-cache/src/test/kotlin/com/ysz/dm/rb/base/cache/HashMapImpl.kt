package com.ysz.dm.rb.base.cache

/**
 * @author carl
 * @create 2022-11-17 9:02 PM
 **/
class HashMapImpl(private val map: MutableMap<String, String> = HashMap(10)) : CustomCache<String, String> {
    override fun get(key: String): String? = map[key]

    override fun multiGet(keys: Collection<String>): Map<String, String> =
        keys.filter { map.containsKey(it) }.associateWith {
            map[it]!!
        }

    override fun invalidate(k: String) {
        map.remove(k)
    }

    override fun put(k: String, v: String) {
        TODO("Not yet implemented")
    }
}