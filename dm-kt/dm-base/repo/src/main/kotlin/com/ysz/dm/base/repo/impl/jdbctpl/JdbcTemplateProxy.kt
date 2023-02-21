package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.repository.InvokeMethodMeta
import com.ysz.dm.base.repo.repository.Repository
import org.springframework.util.ClassUtils
import org.springframework.util.ReflectionUtils
import java.lang.reflect.InvocationHandler
import java.lang.reflect.Method
import java.lang.reflect.Proxy
import kotlin.reflect.full.memberFunctions
import kotlin.reflect.jvm.javaMethod

/**
 * @author carl
 * @since 2023-02-17 3:57 PM
 **/
@Suppress("UNCHECKED_CAST")
class JdbcTemplateProxy<T : Any, ID, I : Repository<T, ID>>(
    private val infJavaClz: Class<I>,
    private val jdbcRepository: SimpleJdbcRepository<T, ID>
) : InvocationHandler {

    private val queries = this.infJavaClz.kotlin.memberFunctions.asSequence()
        .filter { it.javaMethod!!.declaringClass == infJavaClz }
        .map { InvokeMethodMeta.make(it) }
        .associateBy { it.func.javaMethod!! }

    fun getProxy(): I {
        return Proxy.newProxyInstance(
            ClassUtils.getDefaultClassLoader(), arrayOf(
                infJavaClz
            ), this
        ) as I
    }


    override fun invoke(proxy: Any, method: Method, args: Array<out Any>?): Any? {
        return when {/*proxy has no equals & hashCode methods*/
            ReflectionUtils.isEqualsMethod(method) -> equals(args!![0])
            ReflectionUtils.isHashCodeMethod(method) -> hashCode()
            queries.containsKey(method) -> this.jdbcRepository.doInvoke(queries[method]!!, args)
            args == null -> method.invoke(jdbcRepository)
            else -> method.invoke(jdbcRepository, *args)
        }
    }


    override fun toString(): String {
        return "JdbcTplProxy(infJavaClz=$infJavaClz)"
    }

    override fun equals(other: Any?): Boolean {
        return when {
            this === other -> true
            null == other -> false
            other is JdbcTemplateProxy<*, *, *> -> infJavaClz == other.infJavaClz
            else -> false
        }
    }

    override fun hashCode(): Int {
        return infJavaClz.hashCode()
    }

}
