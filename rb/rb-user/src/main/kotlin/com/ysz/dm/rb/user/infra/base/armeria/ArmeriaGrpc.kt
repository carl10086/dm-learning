package com.ysz.dm.rb.user.infra.base.armeria

import org.springframework.core.annotation.AliasFor
import org.springframework.stereotype.Component

/**
 * <pre>
 *     used for application command handler layer
 * </pre>
 * @author carl.yu
 */
@Target(
    AnnotationTarget.TYPE, AnnotationTarget.CLASS
)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
@Component
annotation class ArmeriaGrpc(@get:AliasFor(annotation = Component::class) val value: String = "")