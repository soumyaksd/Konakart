package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetOrderRefunds - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetOrderRefunds
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetOrderRefunds(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminOrderRefundSearchResult getOrderRefunds(String sessionId, AdminOrderRefundSearch retSearch, int offset, int size) throws KKAdminException
     {
         return kkAdminEng.getOrderRefunds(sessionId, retSearch, offset, size);
     }
}
