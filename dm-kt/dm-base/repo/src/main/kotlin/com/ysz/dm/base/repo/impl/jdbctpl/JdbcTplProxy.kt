package com.ysz.dm.base.repo.impl.jdbctpl

import com.ysz.dm.base.repo.repository.Repository
import org.springframework.beans.factory.FactoryBean
import org.springframework.data.util.TypeInformation
import org.springframework.util.ClassUtils
import java.lang.reflect.InvocationHandler
import java.lang.reflect.Method
import java.lang.reflect.Proxy

/**
 * @author carl
 * @since 2023-02-17 3:57 PM
 **/
@Suppress("UNCHECKED_CAST")
class JdbcTplProxy<T>(
    private val infJavaClz: Class<T>,
//    private val jdbcTemplate: JdbcTemplate
) : InvocationHandler {

    fun getProxy(): T {
        val arguments =
            TypeInformation.of(infJavaClz).getRequiredSuperTypeInformation(Repository::class.java).typeArguments

        return Proxy.newProxyInstance(
            ClassUtils.getDefaultClassLoader(),
            arrayOf(
                infJavaClz
            ),
            this
        ) as T
    }


    override fun invoke(proxy: Any, method: Method, args: Array<out Any>): Any {
        return "hello world"
    }

    override fun toString(): String {
        return "JdbcTplProxy(infJavaClz=$infJavaClz)"
    }
}


class JdbcTplProxyFactoryBean<T>(
    private val infJavaClz: Class<T>
) : FactoryBean<T> {
    override fun getObject(): T {
        return JdbcTplProxy(this.infJavaClz).getProxy()
    }

    override fun getObjectType(): Class<*> {
        return infJavaClz
    }

}