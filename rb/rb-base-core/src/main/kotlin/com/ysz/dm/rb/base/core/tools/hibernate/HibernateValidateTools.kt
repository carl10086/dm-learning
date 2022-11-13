package com.ysz.dm.rb.base.core.tools.hibernate

import com.ysz.dm.rb.base.core.exceptions.BaseRbParamException
import org.hibernate.validator.HibernateValidator
import java.util.*
import javax.validation.Validation
import javax.validation.ValidatorFactory

/**
 * @author carl
 */
class HibernateValidateTools {

    companion object {
        private val factory: ValidatorFactory = Validation.byProvider(
            HibernateValidator::class.java
        ).configure() // 快速失败模式
            .failFast(true) // .addProperty( "hibernate.validator.fail_fast", "true" )
            .buildValidatorFactory()
        private val validator = factory.validator

        @Throws(BaseRbParamException::class)
        fun chkThenThrow(bean: Any) {
            chkThenThrow(bean, false)
        }

        @Throws(BaseRbParamException::class)
        fun chkThenThrow(bean: Any, messageWithPath: Boolean) {
            val validateRes = validator!!.validate(bean)
            if (!validateRes.isNullOrEmpty()) {
                val first = validateRes.iterator().next()
                if (first != null) {
                    val path = Objects.toString(first.propertyPath)
                    val message = first.message
                    throw BaseRbParamException(
                        if (messageWithPath) "$path:$message" else "$message"
                    )
                }
            }
        }
    }
}