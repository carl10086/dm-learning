package com.ysz.dm.rb.base.core.ddd

import org.springframework.core.annotation.AliasFor
import org.springframework.stereotype.Component

/**
 * <pre>
 *     when u call other service , u need to adapt it with anti layer
 * </pre>
 * @author carl.yu
 */
@Target(
    AnnotationTarget.ANNOTATION_CLASS,
    AnnotationTarget.CLASS
)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
@Component
annotation class ServiceAdapter(@get:AliasFor(annotation = Component::class) val value: String = "")