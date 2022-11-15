package com.ysz.dm.rb.user.infra.persist.dao

import com.ysz.dm.rb.user.infra.persist.dataobject.UserCoreDO
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.stereotype.Repository

/**
 *<pre>
 * class desc here
 *</pre>
 *@author carl.yu
 *@createAt 2022/11/15
 **/
@Repository
interface UserDao : JpaRepository<UserCoreDO, Long> {
    fun save(mockUser: UserCoreDO)
}