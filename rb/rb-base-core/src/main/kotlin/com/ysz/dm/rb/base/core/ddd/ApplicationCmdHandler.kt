package com.ysz.dm.rb.base.core.ddd

import org.springframework.core.annotation.AliasFor
import org.springframework.stereotype.Component

/**
 * <pre>
 *     used for application command handler layer
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
annotation class ApplicationCmdHandler(@get:AliasFor(annotation = Component::class) val value: String = "")