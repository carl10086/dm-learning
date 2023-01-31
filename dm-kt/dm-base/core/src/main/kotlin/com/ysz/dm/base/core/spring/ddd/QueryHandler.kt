package com.ysz.dm.base.core.spring.ddd

import org.springframework.core.annotation.AliasFor
import org.springframework.stereotype.Component

/**
 * @author carl.yu
 * @since 2022/10/26
 */
@Target(AnnotationTarget.CLASS)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
@Component
annotation class QueryHandler(
    @get:AliasFor(annotation = Component::class) val value: String = "",
)