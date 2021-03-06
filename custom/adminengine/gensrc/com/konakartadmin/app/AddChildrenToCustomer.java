package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - AddChildrenToCustomer - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class AddChildrenToCustomer
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public AddChildrenToCustomer(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public void addChildrenToCustomer(String sessionId, int parentId, AdminCustomer[] children) throws KKAdminException
     {
         kkAdminEng.addChildrenToCustomer(sessionId, parentId, children);
     }
}
