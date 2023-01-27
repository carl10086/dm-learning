package com.ysz.dm.base.core.validate.extend

import jakarta.validation.ConstraintValidator
import jakarta.validation.ConstraintValidatorContext


/**
 * @author carl
 * @create 2022-11-10 5:15 PM
 */
class ValidPhoneValidator : ConstraintValidator<ValidPhone, String> {

    override fun initialize(contactNumber: ValidPhone) {
    }

    override fun isValid(
        contactField: String, cxt: ConstraintValidatorContext
    ): Boolean {
        return PhoneTools.isValidPhone(contactField)
    }
}