package com.ysz.dm.rb.base.core.tools.hibernate.external

import org.passay.*
import javax.validation.ConstraintValidator
import javax.validation.ConstraintValidatorContext

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/10
 **/
class ValidPasswordValidator : ConstraintValidator<ValidPassword, String> {
    override fun isValid(password: String, context: ConstraintValidatorContext): Boolean {
        var validator = PasswordValidator(
            LengthRule(8, 30),
            /*at least 1 uppercase char*/
            CharacterRule(EnglishCharacterData.UpperCase, 1),
            /*at least 1 lowercase char*/
            CharacterRule(EnglishCharacterData.LowerCase, 1),
            /*at least one special*/
            CharacterRule(EnglishCharacterData.Special, 1),
            /*no white space*/
            WhitespaceRule(),
            /*5 sequence like 'abcde' is not permitted*/
            IllegalSequenceRule(EnglishSequenceData.Alphabetical, 5, false)
        )

        var result = validator.validate(PasswordData(password))
        if (result.isValid) {
            return true
        } else {
            context.disableDefaultConstraintViolation()
            context.buildConstraintViolationWithTemplate(
                validator.getMessages(result).joinToString(",")
            ).addConstraintViolation()
            return false;
        }
    }

}