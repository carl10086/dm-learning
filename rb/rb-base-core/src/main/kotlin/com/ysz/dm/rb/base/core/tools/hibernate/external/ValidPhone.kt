package com.ysz.dm.rb.base.core.tools.hibernate.external

import javax.validation.Constraint
import javax.validation.Payload
import kotlin.reflect.KClass

/**
 * @author carl
 * @create 2022-11-10 5:04 PM
 **/
@MustBeDocumented
@Constraint(validatedBy = [ValidPhoneValidator::class])
@Target(
    AnnotationTarget.FUNCTION,
    AnnotationTarget.FIELD
)
@Retention(AnnotationRetention.RUNTIME)
annotation class ValidPhone(
    val message: String = "Invalid phone number",
    val groups: Array<KClass<*>> = [],
    val payload: Array<KClass<out Payload>> = []
) {
}
