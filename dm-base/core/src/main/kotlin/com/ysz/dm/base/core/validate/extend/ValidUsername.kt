package com.ysz.dm.base.core.validate.extend

import jakarta.validation.Constraint
import jakarta.validation.Payload
import kotlin.reflect.KClass

/**
 * @author carl
 * @create 2022-11-10 5:04 PM
 **/
@MustBeDocumented
@Constraint(validatedBy = [ValidUsernameValidator::class])
@Target(
    AnnotationTarget.FUNCTION,
    AnnotationTarget.FIELD
)
@Retention(AnnotationRetention.RUNTIME)
annotation class ValidUsername(
    val message: String = "invalid username",
    val groups: Array<KClass<*>> = [],
    val payload: Array<KClass<out Payload>> = []
) {
}
