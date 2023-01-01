package com.ysz.dm.base.core.validate.extend

import it.unimi.dsi.fastutil.chars.CharOpenHashSet
import it.unimi.dsi.fastutil.chars.CharSet
import jakarta.validation.ConstraintValidator
import jakarta.validation.ConstraintValidatorContext

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/10
 **/
class ValidUsernameValidator : ConstraintValidator<ValidUsername, String> {
    override fun isValid(username: String, context: ConstraintValidatorContext): Boolean {
        if (username.isEmpty() || username.length > MAX_LEN || username.length < MIN_LEN) {
            context.disableDefaultConstraintViolation()
            context.buildConstraintViolationWithTemplate(LENGTH_ERROR_MSG).addConstraintViolation()
            return false;
        }

        for (aChar in username) {
            if (!validChar(aChar)) {
                context.disableDefaultConstraintViolation()
                context.buildConstraintViolationWithTemplate(
                    CHAR_ERROR_MSG
                ).addConstraintViolation()
                return false;
            }
        }

        return true;
    }


    companion object {
        private const val MAX_LEN = 30;
        private const val MIN_LEN = 10;
        private val validCharSet: CharSet;
        const val LENGTH_ERROR_MSG = "username length must between $MIN_LEN and $MAX_LEN"
        const val CHAR_ERROR_MSG = "username can only be 'a..zA..Z0..9_-"


        fun validChar(aChar: Char): Boolean {
            return validCharSet.contains(aChar)
        }


        init {

            val specialChars = charArrayOf('-', '_')
            var charSet = CharOpenHashSet(26/*a->z*/ + 26/*A->Z*/ + 10/*0-9*/ + specialChars.size/*-_*/)
            for (i in 0..25) charSet.add('a' + i)
            for (i in 0..25) charSet.add('A' + i)
            for (i in 0..9) charSet.add('0' + i)

            for (aChar in specialChars) charSet.add(aChar)

            this.validCharSet = charSet


        }
    }


}