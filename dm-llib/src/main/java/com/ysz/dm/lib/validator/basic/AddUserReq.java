package com.ysz.dm.lib.validator.basic;


import javax.validation.constraints.NotNull;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/11/9
 **/
public record AddUserReq(
    @NotNull(message = "username can't be null")
    String username,
    @ValidPassword
    String password
) {

}
