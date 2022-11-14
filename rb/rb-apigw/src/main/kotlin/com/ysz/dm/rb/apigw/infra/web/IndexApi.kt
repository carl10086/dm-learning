package com.ysz.dm.rb.apigw.infra.web

import com.ysz.dm.rb.apigw.infra.web.security.Roles
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RestController
import javax.annotation.security.RolesAllowed
import javax.servlet.http.HttpServletRequest

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/12
 **/
@RestController
@RequestMapping("/index")
open class IndexApi {


    @GetMapping("/hello")
    fun hello() = "hello security"


    @PostMapping("/post")
    @RolesAllowed(Roles.ADMIN)
    open fun post(
        req: HttpServletRequest
    ): String {
        var principal = req.userPrincipal
        return "ok"
    }
}