package com.ysz.dm.base.core.validate.extend

import com.google.i18n.phonenumbers.PhoneNumberUtil
import com.google.i18n.phonenumbers.Phonenumber

/**
 * @author carl
 * @create 2022-11-10 5:38 PM
 **/
class PhoneTools {

    companion object {
        /**
         * check valid phone
         */
        fun isValidPhone(src: String): Boolean {
            return isValidPhone(src, "CN")
        }

        /**
         * @param src: source string
         * @param country: country locale, {@link 参考: https://www.iso.org/iso-3166-country-codes.html}
         */
        private fun isValidPhone(src: String, country: String): Boolean {
            if (src.isNullOrBlank()) {
                return false;
            }
            val ins = PhoneNumberUtil.getInstance()
            var number: Phonenumber.PhoneNumber?
            try {
                number = ins.parse(src, country)
            } catch (ignored: Exception) {
                return false
            }

            return ins.isValidNumber(number)
        }
    }
}