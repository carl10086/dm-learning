package com.ysz.dm.base.repo.repository

import com.ysz.dm.base.core.domain.page.Page
import com.ysz.dm.base.core.domain.page.PageRequest
import com.ysz.dm.base.repo.support.query.PartTree
import kotlin.reflect.KClass
import kotlin.reflect.KFunction
import kotlin.reflect.full.isSuperclassOf
import kotlin.reflect.jvm.jvmErasure

/**
 * @author carl
 * @since 2023-02-21 12:52 AM
 **/
data class InvokeMethodMeta(
    /*function */
    val func: KFunction<*>,
    /*partTree*/
    val partTree: PartTree,
    /*if has pageAbleArg, this is index*/
    val pageAbleArgIndex: Int = -1,
    val resultType: ResultTypeMeta
) {
    companion object {
        fun make(func: KFunction<*>): InvokeMethodMeta {
            val partTree = PartTree(func.name)
            val pageArgList = func.parameters
                .asSequence()
                .withIndex()
                .filter { PageRequest::class.isSuperclassOf(it.value.type.jvmErasure) }
                .toList()

            val pageAbleArgIndex = when (pageArgList.size) {
                0 -> -1
                1 -> pageArgList.first().index - 1 /*first args always is `this`*/
                else -> throw IllegalArgumentException("page request arg numbers must <= 1")
            }

            val resultTypeMeta = ResultTypeMeta.make(func.returnType.jvmErasure)

            check(pageAbleArgIndex < 0 || resultTypeMeta.isPage) { "pagination method returnType must implement Page interface" }

            val subject = partTree.subject
            when {
                subject.count -> check(resultTypeMeta.isIntOrLong()) { "count method must return int or long " }
                subject.exists -> check(resultTypeMeta.isBoolean) { "exists method must return boolean" }
                subject.delete -> check(resultTypeMeta.isIntOrLong()) { "delete method must return int or long" }
                resultTypeMeta.isCollection -> check(List::class.isSuperclassOf(resultTypeMeta.kClass)){"collection now only supprt list"}
                else -> {}
            }

            return InvokeMethodMeta(func, partTree, pageAbleArgIndex, resultTypeMeta)
        }
    }
}

data class ResultTypeMeta(
    val kClass: KClass<*>,
    val isCollection: Boolean = false,
    val isArray: Boolean = false,
    val isPage: Boolean = false,
    val isBoolean: Boolean = false,
    val isInt: Boolean = false,
    val isLong: Boolean = false
) {
    fun isIntOrLong() = this.isInt || this.isLong

    companion object {
        fun make(kClass: KClass<*>): ResultTypeMeta {
            return when {
                kClass == Long::class -> ResultTypeMeta(kClass, isLong = true)
                kClass == Int::class -> ResultTypeMeta(kClass, isInt = true)
                kClass == Boolean::class -> ResultTypeMeta(kClass, isBoolean = true)
                Collection::class.isSuperclassOf(kClass) -> ResultTypeMeta(kClass, isCollection = true)
                kClass.java.isArray -> ResultTypeMeta(kClass, isArray = true)
                Page::class.isSuperclassOf(kClass) -> ResultTypeMeta(kClass, isPage = true)
                else -> ResultTypeMeta(kClass)
            }
        }
    }
}