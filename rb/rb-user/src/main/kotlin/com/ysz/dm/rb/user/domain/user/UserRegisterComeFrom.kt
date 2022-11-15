package com.ysz.dm.rb.user.domain.user

/**
 * this class represents where user register from
 * @author carl
 * @create 2022-11-10 6:47 PM
 **/
enum class UserRegisterComeFrom(val code: Int) {
    UNKNOWN(-1),

    /**
     * origin web hook
     */
    RB_ORIGINAL_WEB(0)
    ;

    companion object {
        fun of(code: Int): UserRegisterComeFrom {
            for (item in values()) {
                if (item.code == code) {
                    return item
                }
            }

            return UNKNOWN;
        }
    }
}