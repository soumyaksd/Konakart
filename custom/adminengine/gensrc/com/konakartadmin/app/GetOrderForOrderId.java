package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetOrderForOrderId - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetOrderForOrderId
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetOrderForOrderId(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminOrder getOrderForOrderId(String sessionId, int orderId) throws KKAdminException
     {
         return kkAdminEng.getOrderForOrderId(sessionId, orderId);
     }
}