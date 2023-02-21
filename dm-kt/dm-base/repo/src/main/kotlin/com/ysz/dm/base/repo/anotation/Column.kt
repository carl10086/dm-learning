package com.ysz.dm.base.repo.anotation

/**
 * @author carl
 * @since 2023-02-21 12:37 PM
 **/
@Retention(AnnotationRetention.RUNTIME)
@Target(AnnotationTarget.FIELD)
annotation class Column(
    val name: String
)
