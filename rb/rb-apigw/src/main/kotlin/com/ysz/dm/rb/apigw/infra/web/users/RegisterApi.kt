package com.ysz.dm.rb.apigw.infra.web.users

import com.ysz.dm.rb.apigw.infra.web.dto.common.ApiResp
import com.ysz.dm.rb.apigw.infra.web.users.req.CheckUsernameReq
import com.ysz.dm.rb.base.core.tools.hibernate.HibernateValidateTools
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/13
 **/
@RestController
@RequestMapping("/users/register")
class RegisterApi {


    @PostMapping("/checks/username")
    fun checkUsername(req: CheckUsernameReq): ApiResp<Boolean> {
        return ApiResp(HibernateValidateTools.chkSuccess(req))
    }


}