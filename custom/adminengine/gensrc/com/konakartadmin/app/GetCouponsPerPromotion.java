package com.konakartadmin.app;

import com.konakartadmin.appif.*;
import com.konakartadmin.bl.KKAdmin;

/**
 *  The KonaKart Custom Engine - GetCouponsPerPromotion - Generated by CreateKKAdminCustomEng
 */
@SuppressWarnings("all")
public class GetCouponsPerPromotion
{
    KKAdmin kkAdminEng = null;

    /**
     * Constructor
     */
     public GetCouponsPerPromotion(KKAdmin _kkAdminEng)
     {
         kkAdminEng = _kkAdminEng;
     }

     public AdminCoupon[] getCouponsPerPromotion(String sessionId, int promotionId) throws KKAdminException
     {
         return kkAdminEng.getCouponsPerPromotion(sessionId, promotionId);
     }
}
