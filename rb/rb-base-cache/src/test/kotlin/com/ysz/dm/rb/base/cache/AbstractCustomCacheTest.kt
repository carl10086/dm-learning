package com.ysz.dm.rb.base.cache//package com.ysz.dm.rb
//
//import org.junit.jupiter.api.Assertions
//import org.junit.jupiter.api.Test
//import java.util.function.Function
//
///**
// * <pre>
// * class desc here
//</pre> *
// * @author carl.yu
// * @createAt 2022/11/16
// */
//internal class AbstractCustomCacheTest {
//
//    private val cfg = CustomCacheCfg(PREFIX, "")
//
//    private val hashMapImpl = HashMapImpl(cfg = cfg)
//    private val alwaysFailImpl = AlwaysFailImpl(cfg)
//
//
//    @Test
//    internal fun `test_get`() {
//        /*k1 is already existed*/
//        Assertions.assertEquals(hashMapImpl.get("k1", Function.identity()), "v1")
//        /*k2 use identity loader*/
//        Assertions.assertEquals(hashMapImpl.get("k2", Function.identity()), "k2")
//        /*k2 use null loader*/
//        Assertions.assertEquals(hashMapImpl.get("k2", { null }), "k2")
//        /*k3 use null loader*/
//        Assertions.assertEquals(hashMapImpl.get("k3", { null }), null)
//
//        /*test fail without fallback*/
//        Assertions.assertThrows(
//            CustomCacheException::class.java,
//        ) { alwaysFailImpl.get("k1", Function.identity(), fallback = null) }
//
//        /*test fail with fallback*/
//        Assertions.assertEquals("fallback", alwaysFailImpl.get("k1", Function.identity(), fallback = { "fallback" }))
//    }
//
//
//    @Test
//    internal fun `test_multiGet`() {
//        /*k1 from cache while k2  from sor, k3 no data*/
//        val values = hashMapImpl.multiGet(
//            listOf("k1", "k2", "k3"),
//            { keys -> keys.filter { it -> it == "k2" }.associateBy { it } })
//
//        Assertions.assertEquals(values["k1"], "v1")
//        Assertions.assertEquals(values["k2"], "k2")
//        Assertions.assertEquals(values["k3"], null)
//    }
//
//    /**
//     * always fail action
//     */
//    internal class AlwaysFailImpl(
//        cfg: CustomCacheCfg
//    ) :
//        AbstractCustomCache<String, String>(cfg) {
//        override fun doGet(k: String) = throw CustomCacheException("fail")
//        override fun doPut(k: String, v: String) = throw CustomCacheException("fail")
//        override fun doMultiGet(keys: Collection<String>): Map<String, String> = throw CustomCacheException("fail")
//        override fun doMultiPut(map: Map<String, String>) = throw CustomCacheException("fail")
//        override fun rm(k: String): Boolean = throw CustomCacheException("fail")
//    }
//
//
//    /**
//     * hash map implementation
//     */
//    internal class HashMapImpl(
//        private val map: MutableMap<String, String> = mutableMapOf(Pair("k1", "v1")),
//        cfg: CustomCacheCfg
//    ) :
//        AbstractCustomCache<String, String>(cfg) {
//        override fun doGet(k: String) = map[k]
//
//        override fun doPut(k: String, v: String) {
//            map[k] = v
//        }
//
//        override fun doMultiGet(keys: Collection<String>): Map<String, String> {
//            return keys.filter { map.containsKey(it) }.associateWith { map[it]!! }
//        }
//
//        override fun rm(k: String): Boolean {
//            TODO("Not yet implemented")
//        }
//    }
//
//    companion object {
//        private const val PREFIX = "test:"
//    }
//}