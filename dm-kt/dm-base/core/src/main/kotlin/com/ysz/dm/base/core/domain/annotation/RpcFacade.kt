package com.ysz.dm.base.core.domain.annotation

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
annotation class RpcFacade(
    @get:AliasFor(annotation = Component::class) val value: String = "",
)